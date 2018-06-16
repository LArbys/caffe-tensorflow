from kaffe.tensorflow import Network

class UResNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 16, 1, 1, relu=False, name='conv0')
             .batch_normalization(relu=True, name='bn_conv0')
             .max_pool(3, 3, 2, 2, name='pool0')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='res1a_branch1')
             .batch_normalization(name='bn1a_branch1'))

        (self.feed('pool0')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='res1a_branch2a')
             .batch_normalization(relu=True, name='bn1a_branch2a')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='res1a_branch2b')
             .batch_normalization(relu=True, name='bn1a_branch2b'))

        (self.feed('bn1a_branch1', 
                   'bn1a_branch2b')
             .add(name='res1a')
             .relu(name='res1a_relu')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='res1b_branch2a')
             .batch_normalization(relu=True, name='bn1b_branch2a')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='res1b_branch2b')
             .batch_normalization(relu=True, name='bn1b_branch2b'))

        (self.feed('res1a_relu', 
                   'bn1b_branch2b')
             .add(name='res1b')
             .relu(name='res1b_relu')
             .conv(1, 1, 64, 2, 2, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(name='bn2a_branch1'))

        (self.feed('res1b_relu')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(relu=True, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(relu=True, name='bn2a_branch2b'))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2b')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(relu=True, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(relu=True, name='bn2b_branch2b'))

        (self.feed('res2a_relu', 
                   'bn2b_branch2b')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(name='bn3a_branch1'))

        (self.feed('res2b_relu')
             .conv(3, 3, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(relu=True, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(relu=True, name='bn3a_branch2b'))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2b')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
             .batch_normalization(relu=True, name='bn3b_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
             .batch_normalization(relu=True, name='bn3b_branch2b'))

        (self.feed('res3a_relu', 
                   'bn3b_branch2b')
             .add(name='res3b')
             .relu(name='res3b_relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(name='bn4a_branch1'))

        (self.feed('res3b_relu')
             .conv(3, 3, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(relu=True, name='bn4a_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(relu=True, name='bn4a_branch2b'))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2b')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
             .batch_normalization(relu=True, name='bn4b_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b_branch2b')
             .batch_normalization(relu=True, name='bn4b_branch2b'))

        (self.feed('res4a_relu', 
                   'bn4b_branch2b')
             .add(name='res4b')
             .relu(name='res4b_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(name='bn5a_branch1'))

        (self.feed('res4b_relu')
             .conv(3, 3, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(relu=True, name='bn5a_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(relu=True, name='bn5a_branch2b'))

        (self.feed('bn5a_branch1', 
                   'bn5a_branch2b')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(relu=True, name='bn5b_branch2a')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(relu=True, name='bn5b_branch2b'))

        (self.feed('res5a_relu', 
                   'bn5b_branch2b')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .deconv(4, 4, 256, 2, 2, group=256, relu=False, name='deconv0_deconv'))

        (self.feed('res4b_relu', 
                   'deconv0_deconv')
             .concat(3, name='deconv0_concat')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res6a_branch1')
             .batch_normalization(name='bn6a_branch1'))

        (self.feed('deconv0_concat')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res6a_branch2a')
             .batch_normalization(relu=True, name='bn6a_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res6a_branch2b')
             .batch_normalization(relu=True, name='bn6a_branch2b'))

        (self.feed('bn6a_branch1', 
                   'bn6a_branch2b')
             .add(name='res6a')
             .relu(name='res6a_relu')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res6b_branch2a')
             .batch_normalization(relu=True, name='bn6b_branch2a')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res6b_branch2b')
             .batch_normalization(relu=True, name='bn6b_branch2b'))

        (self.feed('res6a_relu', 
                   'bn6b_branch2b')
             .add(name='res6b')
             .relu(name='res6b_relu')
             .deconv(4, 4, 128, 2, 2, group=128, relu=False, name='deconv1_deconv'))

        (self.feed('res3b_relu', 
                   'deconv1_deconv')
             .concat(3, name='deconv1_concat')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res7a_branch1')
             .batch_normalization(name='bn7a_branch1'))

        (self.feed('deconv1_concat')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res7a_branch2a')
             .batch_normalization(relu=True, name='bn7a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res7a_branch2b')
             .batch_normalization(relu=True, name='bn7a_branch2b'))

        (self.feed('bn7a_branch1', 
                   'bn7a_branch2b')
             .add(name='res7a')
             .relu(name='res7a_relu')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res7b_branch2a')
             .batch_normalization(relu=True, name='bn7b_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res7b_branch2b')
             .batch_normalization(relu=True, name='bn7b_branch2b'))

        (self.feed('res7a_relu', 
                   'bn7b_branch2b')
             .add(name='res7b')
             .relu(name='res7b_relu')
             .deconv(4, 4, 64, 2, 2, group=64, relu=False, name='deconv2_deconv'))

        (self.feed('res2b_relu', 
                   'deconv2_deconv')
             .concat(3, name='deconv2_concat')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res8a_branch1')
             .batch_normalization(name='bn8a_branch1'))

        (self.feed('deconv2_concat')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res8a_branch2a')
             .batch_normalization(relu=True, name='bn8a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res8a_branch2b')
             .batch_normalization(relu=True, name='bn8a_branch2b'))

        (self.feed('bn8a_branch1', 
                   'bn8a_branch2b')
             .add(name='res8a')
             .relu(name='res8a_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res8b_branch2a')
             .batch_normalization(relu=True, name='bn8b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res8b_branch2b')
             .batch_normalization(relu=True, name='bn8b_branch2b'))

        (self.feed('res8a_relu', 
                   'bn8b_branch2b')
             .add(name='res8b')
             .relu(name='res8b_relu')
             .deconv(4, 4, 32, 2, 2, group=32, relu=False, name='deconv3_deconv'))

        (self.feed('res1b_relu', 
                   'deconv3_deconv')
             .concat(3, name='deconv3_concat')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='res9a_branch1')
             .batch_normalization(name='bn9a_branch1'))

        (self.feed('deconv3_concat')
             .conv(5, 5, 32, 1, 1, biased=False, relu=False, name='res9a_branch2a')
             .batch_normalization(relu=True, name='bn9a_branch2a')
             .conv(5, 5, 32, 1, 1, biased=False, relu=False, name='res9a_branch2b')
             .batch_normalization(relu=True, name='bn9a_branch2b'))

        (self.feed('bn9a_branch1', 
                   'bn9a_branch2b')
             .add(name='res9a')
             .relu(name='res9a_relu')
             .conv(5, 5, 32, 1, 1, biased=False, relu=False, name='res9b_branch2a')
             .batch_normalization(relu=True, name='bn9b_branch2a')
             .conv(5, 5, 32, 1, 1, biased=False, relu=False, name='res9b_branch2b')
             .batch_normalization(relu=True, name='bn9b_branch2b'))

        (self.feed('res9a_relu', 
                   'bn9b_branch2b')
             .add(name='res9b')
             .relu(name='res9b_relu')
             .deconv(4, 4, 16, 2, 2, group=16, relu=False, name='deconv4_deconv'))

        (self.feed('deconv4_deconv', 
                   'bn_conv0')
             .concat(3, name='deconv4_concat')
             .conv(7, 7, 16, 1, 1, relu=False, name='conv10')
             .batch_normalization(relu=True, name='bn_conv10')
             .conv(7, 7, 3, 1, 1, relu=False, name='conv11')
             .batch_normalization(relu=True, name='bn_conv11')
             .softmax(name='softmax'))